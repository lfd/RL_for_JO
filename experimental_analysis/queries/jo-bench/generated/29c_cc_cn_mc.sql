SELECT * FROM company_name AS cn, complete_cast AS cc, movie_companies AS mc WHERE cn.country_code = '[us]' AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id;