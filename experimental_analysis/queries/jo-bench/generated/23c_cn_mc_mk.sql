SELECT * FROM company_name AS cn, movie_keyword AS mk, movie_companies AS mc WHERE cn.country_code = '[us]' AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id;