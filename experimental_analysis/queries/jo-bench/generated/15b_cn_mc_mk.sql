SELECT * FROM movie_keyword AS mk, company_name AS cn, movie_companies AS mc WHERE cn.country_code = '[us]' AND cn.name = 'YouTube' AND mc.note LIKE '%(200%)%' AND mc.note LIKE '%(worldwide)%' AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id;